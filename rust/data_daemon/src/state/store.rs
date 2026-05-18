//! SQLite-backed implementation of the daemon's [`StateStore`].
//!
//! Phase 3 establishes the persistence layer: schema migration via `sqlx`, WAL
//! pragmas on every connection, and the core CRUD operations the per-trace
//! actors and registration coordinator need in Phase 4+. The trait surface is
//! intentionally narrower than today's Python `StateStore` Protocol; methods
//! get added as later phases need them, and the schema may evolve in lockstep
//! (see `docs/data-daemon-rewrite.md` §5: "Schema is flexible.").

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};
use sqlx::{ConnectOptions, SqliteConnection, SqlitePool};
use thiserror::Error;
use tokio::sync::Mutex;

use crate::state::schema::{
    RecordingRow, TraceRecord, TraceRegistrationStatus, TraceUploadStatus, TraceWriteStatus,
};

/// Embedded migrations, applied on every [`SqliteStateStore::open`].
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Busy timeout applied to every connection, matching the Python
/// `PRAGMA busy_timeout=1000` in `state_store_sqlite.py::_apply_pragmas`.
const BUSY_TIMEOUT_MS: u32 = 1000;

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

/// Persistence interface for daemon state.
///
/// Phase 3 implements the slice of the Python `StateStore` protocol needed for
/// the dispatcher + per-trace actor bring-up in Phase 4. Additional methods —
/// upload bookkeeping, retry scheduling, reconciliation — are added in the
/// phases that need them.
#[async_trait]
pub trait StateStore: Send + Sync {
    /// Insert a recording row if one does not already exist.
    async fn create_recording(&self, recording_id: &str) -> Result<RecordingRow, StateStoreError>;

    /// Fetch a recording by ID, returning `None` when absent.
    async fn get_recording(
        &self,
        recording_id: &str,
    ) -> Result<Option<RecordingRow>, StateStoreError>;

    /// Insert a trace row in the [`TraceWriteStatus::Initializing`] state.
    ///
    /// The parent recording is created on demand to mirror the Python
    /// `state_manager` behaviour where a `START_TRACE` may race ahead of any
    /// explicit recording creation.
    async fn create_trace(
        &self,
        recording_id: &str,
        trace_id: &str,
        data_type: Option<&str>,
        data_type_name: Option<&str>,
    ) -> Result<TraceRecord, StateStoreError>;

    /// Apply a partial update to an existing trace.
    ///
    /// Only set fields are written; unset fields preserve their existing
    /// value. Returns the trace row after the update, or `None` if the row
    /// does not exist.
    async fn update_trace(
        &self,
        trace_id: &str,
        update: TraceUpdate,
    ) -> Result<Option<TraceRecord>, StateStoreError>;

    /// Fetch a trace by ID, returning `None` when absent.
    async fn get_trace(&self, trace_id: &str) -> Result<Option<TraceRecord>, StateStoreError>;

    /// Return all traces for the given recording, ordered by `created_at`.
    async fn list_traces_for_recording(
        &self,
        recording_id: &str,
    ) -> Result<Vec<TraceRecord>, StateStoreError>;

    /// Claim up to `limit` traces in [`TraceWriteStatus::Written`] /
    /// [`TraceRegistrationStatus::Pending`] for registration.
    ///
    /// Traces are eligible immediately when at least `limit` are ready (size
    /// trigger) or when their `last_updated` is older than `max_wait_secs`
    /// (age trigger) — matching the Python registration coordinator's
    /// debounce policy described in §4 of the rewrite plan.
    ///
    /// Claimed rows are transitioned to
    /// [`TraceRegistrationStatus::Registering`] atomically inside a single
    /// transaction so two coordinators cannot double-claim.
    async fn claim_traces_for_registration(
        &self,
        limit: usize,
        max_wait_secs: f64,
    ) -> Result<Vec<TraceRecord>, StateStoreError>;

    /// Mark a recording as stopped by setting its `stopped_at` to now.
    ///
    /// Idempotent: re-stopping a recording that already has a `stopped_at`
    /// leaves the existing timestamp untouched so duplicate `StopRecording`
    /// envelopes (e.g. SDK retry on socket reconnect) do not slide the wall
    /// time forward. The recording row is created on demand if it does not
    /// already exist so a stop that races ahead of `create_recording` still
    /// records the terminal state.
    async fn mark_recording_stopped(
        &self,
        recording_id: &str,
    ) -> Result<RecordingRow, StateStoreError>;
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
    }
}

/// SQLite-backed [`StateStore`].
///
/// Writes are serialised through a `Mutex<()>` mirroring the
/// `asyncio.Semaphore(1)` guard in `state_store_sqlite.py:63`. Reads run in
/// parallel through the underlying pool.
#[derive(Clone)]
pub struct SqliteStateStore {
    pool: SqlitePool,
    write_guard: Arc<Mutex<()>>,
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

        let mut options = SqliteConnectOptions::new()
            .filename(db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .busy_timeout(std::time::Duration::from_millis(BUSY_TIMEOUT_MS as u64));
        // sqlx prints every statement at INFO by default; quiet that down so
        // the daemon's tracing output isn't drowned out by the same SQL on
        // every trace write.
        options = options.log_statements(tracing::log::LevelFilter::Debug);

        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .connect_with(options)
            .await?;

        MIGRATOR.run(&pool).await?;

        Ok(SqliteStateStore {
            pool,
            write_guard: Arc::new(Mutex::new(())),
        })
    }

    /// Borrow the underlying pool, e.g. for diagnostics in tests.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Close the pool, draining outstanding connections.
    pub async fn close(self) {
        self.pool.close().await;
    }

    async fn upsert_recording_locked(
        conn: &mut SqliteConnection,
        recording_id: &str,
    ) -> Result<RecordingRow, sqlx::Error> {
        let now = Utc::now().naive_utc();
        sqlx::query(
            "INSERT INTO recordings (recording_id, created_at, last_updated) \
             VALUES (?1, ?2, ?2) \
             ON CONFLICT(recording_id) DO NOTHING",
        )
        .bind(recording_id)
        .bind(now)
        .execute(&mut *conn)
        .await?;

        let row = sqlx::query("SELECT * FROM recordings WHERE recording_id = ?1")
            .bind(recording_id)
            .fetch_one(&mut *conn)
            .await?;
        RecordingRow::from_row(&row)
    }
}

#[async_trait]
impl StateStore for SqliteStateStore {
    async fn create_recording(&self, recording_id: &str) -> Result<RecordingRow, StateStoreError> {
        let _guard = self.write_guard.lock().await;
        let mut tx = self.pool.begin().await?;
        let row = Self::upsert_recording_locked(&mut tx, recording_id).await?;
        tx.commit().await?;
        Ok(row)
    }

    async fn get_recording(
        &self,
        recording_id: &str,
    ) -> Result<Option<RecordingRow>, StateStoreError> {
        let row = sqlx::query("SELECT * FROM recordings WHERE recording_id = ?1")
            .bind(recording_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(match row {
            Some(row) => Some(RecordingRow::from_row(&row)?),
            None => None,
        })
    }

    async fn create_trace(
        &self,
        recording_id: &str,
        trace_id: &str,
        data_type: Option<&str>,
        data_type_name: Option<&str>,
    ) -> Result<TraceRecord, StateStoreError> {
        let _guard = self.write_guard.lock().await;
        let mut tx = self.pool.begin().await?;

        Self::upsert_recording_locked(&mut tx, recording_id).await?;

        let now = Utc::now().naive_utc();
        sqlx::query(
            "INSERT INTO traces (trace_id, recording_id, write_status, data_type, \
                                 data_type_name, created_at, last_updated) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) \
             ON CONFLICT(trace_id) DO NOTHING",
        )
        .bind(trace_id)
        .bind(recording_id)
        .bind(TraceWriteStatus::Initializing.as_str())
        .bind(data_type)
        .bind(data_type_name)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        // Keep the parent recording's trace_count in step with what we observe
        // so the upload coordinator can read it without a `COUNT(*)` later.
        sqlx::query(
            "UPDATE recordings \
                SET trace_count = (SELECT COUNT(*) FROM traces WHERE recording_id = ?1), \
                    last_updated = ?2 \
              WHERE recording_id = ?1",
        )
        .bind(recording_id)
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
    ) -> Result<Option<TraceRecord>, StateStoreError> {
        if update.is_empty() {
            return self.get_trace(trace_id).await;
        }

        let _guard = self.write_guard.lock().await;
        let mut tx = self.pool.begin().await?;

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
        query = query.bind(now).bind(trace_id);

        let result = query.execute(&mut *tx).await?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let row = sqlx::query("SELECT * FROM traces WHERE trace_id = ?1")
            .bind(trace_id)
            .fetch_one(&mut *tx)
            .await?;
        let record = TraceRecord::from_row(&row)?;

        tx.commit().await?;
        Ok(Some(record))
    }

    async fn get_trace(&self, trace_id: &str) -> Result<Option<TraceRecord>, StateStoreError> {
        let row = sqlx::query("SELECT * FROM traces WHERE trace_id = ?1")
            .bind(trace_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(match row {
            Some(row) => Some(TraceRecord::from_row(&row)?),
            None => None,
        })
    }

    async fn list_traces_for_recording(
        &self,
        recording_id: &str,
    ) -> Result<Vec<TraceRecord>, StateStoreError> {
        let rows = sqlx::query(
            "SELECT * FROM traces WHERE recording_id = ?1 ORDER BY created_at ASC, trace_id ASC",
        )
        .bind(recording_id)
        .fetch_all(&self.pool)
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
        let _guard = self.write_guard.lock().await;
        let mut tx = self.pool.begin().await?;

        // Count ready traces first so the size-vs-age policy from §4 of the
        // plan stays explicit. SQLite's transactional snapshot means this
        // count is stable across the subsequent SELECT.
        let ready_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM traces \
              WHERE write_status = ?1 AND registration_status = ?2",
        )
        .bind(TraceWriteStatus::Written.as_str())
        .bind(TraceRegistrationStatus::Pending.as_str())
        .fetch_one(&mut *tx)
        .await?;

        let limit_i64 = limit as i64;
        let rows = if ready_count >= limit_i64 && limit_i64 > 0 {
            sqlx::query(
                "SELECT * FROM traces \
                  WHERE write_status = ?1 AND registration_status = ?2 \
               ORDER BY last_updated ASC \
                  LIMIT ?3",
            )
            .bind(TraceWriteStatus::Written.as_str())
            .bind(TraceRegistrationStatus::Pending.as_str())
            .bind(limit_i64)
            .fetch_all(&mut *tx)
            .await?
        } else {
            let cutoff = Utc::now().naive_utc()
                - chrono::Duration::milliseconds((max_wait_secs * 1000.0) as i64);
            sqlx::query(
                "SELECT * FROM traces \
                  WHERE write_status = ?1 AND registration_status = ?2 \
                    AND last_updated <= ?3 \
               ORDER BY last_updated ASC \
                  LIMIT ?4",
            )
            .bind(TraceWriteStatus::Written.as_str())
            .bind(TraceRegistrationStatus::Pending.as_str())
            .bind(cutoff)
            .bind(limit_i64)
            .fetch_all(&mut *tx)
            .await?
        };

        let mut claimed = Vec::with_capacity(rows.len());
        let now = Utc::now().naive_utc();
        for row in &rows {
            let trace = TraceRecord::from_row(row)?;
            sqlx::query(
                "UPDATE traces SET registration_status = ?1, last_updated = ?2 \
                  WHERE trace_id = ?3 AND registration_status = ?4",
            )
            .bind(TraceRegistrationStatus::Registering.as_str())
            .bind(now)
            .bind(&trace.trace_id)
            .bind(TraceRegistrationStatus::Pending.as_str())
            .execute(&mut *tx)
            .await?;
            claimed.push(TraceRecord {
                registration_status: TraceRegistrationStatus::Registering,
                last_updated: now,
                ..trace
            });
        }

        tx.commit().await?;
        Ok(claimed)
    }

    async fn mark_recording_stopped(
        &self,
        recording_id: &str,
    ) -> Result<RecordingRow, StateStoreError> {
        let _guard = self.write_guard.lock().await;
        let mut tx = self.pool.begin().await?;
        // Ensure the recording row exists — the SDK may publish StopRecording
        // before the StartRecording envelope under load.
        Self::upsert_recording_locked(&mut tx, recording_id).await?;

        let now = Utc::now().naive_utc();
        sqlx::query(
            "UPDATE recordings \
                SET stopped_at = COALESCE(stopped_at, ?2), \
                    last_updated = ?2 \
              WHERE recording_id = ?1",
        )
        .bind(recording_id)
        .bind(now)
        .execute(&mut *tx)
        .await?;

        let row = sqlx::query("SELECT * FROM recordings WHERE recording_id = ?1")
            .bind(recording_id)
            .fetch_one(&mut *tx)
            .await?;
        let record = RecordingRow::from_row(&row)?;

        tx.commit().await?;
        Ok(record)
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
    async fn create_trace_inserts_recording_and_trace_rows() {
        let (store, _tempdir) = open_store().await;

        let trace = store
            .create_trace("rec-1", "trace-1", Some("video"), None)
            .await
            .expect("create_trace");
        assert_eq!(trace.trace_id, "trace-1");
        assert_eq!(trace.recording_id, "rec-1");
        assert_eq!(trace.write_status, TraceWriteStatus::Initializing);
        assert_eq!(trace.data_type.as_deref(), Some("video"));

        let recording = store
            .get_recording("rec-1")
            .await
            .expect("get_recording")
            .expect("recording row");
        assert_eq!(recording.trace_count, 1);

        // Creating the same trace twice is a no-op (write_status preserved).
        let again = store
            .create_trace("rec-1", "trace-1", Some("video"), None)
            .await
            .expect("idempotent create_trace");
        assert_eq!(again.trace_id, "trace-1");
        let traces = store
            .list_traces_for_recording("rec-1")
            .await
            .expect("list_traces");
        assert_eq!(traces.len(), 1);
    }

    #[tokio::test]
    async fn update_trace_overwrites_only_set_fields() {
        let (store, _tempdir) = open_store().await;
        store
            .create_trace("rec-1", "trace-1", None, None)
            .await
            .expect("create_trace");

        let updated = store
            .update_trace(
                "trace-1",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Writing),
                    bytes_written: Some(2048),
                    ..TraceUpdate::default()
                },
            )
            .await
            .expect("update_trace")
            .expect("trace exists");
        assert_eq!(updated.write_status, TraceWriteStatus::Writing);
        assert_eq!(updated.bytes_written, 2048);
        // Unset fields keep their prior values.
        assert_eq!(updated.bytes_uploaded, 0);
        assert_eq!(updated.upload_status, TraceUploadStatus::Pending);

        let missing = store
            .update_trace(
                "unknown-trace",
                TraceUpdate {
                    write_status: Some(TraceWriteStatus::Failed),
                    ..TraceUpdate::default()
                },
            )
            .await
            .expect("update_trace");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn claim_for_registration_respects_size_trigger() {
        let (store, _tempdir) = open_store().await;
        for index in 0..5 {
            let trace_id = format!("trace-{index}");
            store
                .create_trace("rec-1", &trace_id, None, None)
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
    async fn claim_for_registration_respects_age_trigger() {
        let (store, _tempdir) = open_store().await;
        store
            .create_trace("rec-1", "trace-1", None, None)
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
}
