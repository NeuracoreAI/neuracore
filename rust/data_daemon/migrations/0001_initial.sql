-- Initial schema for the Rust data daemon state store.
--
-- The daemon owns recording/trace identity: `recordings` are keyed by a local
-- autoincrement `recording_index` (the cloud `recording_id` is backfilled
-- asynchronously by the start notifier), and `traces` are keyed by a
-- daemon-minted UUID. Column names and the status-enum strings are part of the
-- behavioural contract the integration suite relies on (see
-- tests/integration/platform/data_daemon/shared/db_constants.py).

CREATE TABLE IF NOT EXISTS recordings (
    recording_index               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Cloud handle. NULL until the recording-start notifier POSTs
    -- `/recording/start`.
    recording_id                  TEXT,
    robot_id                      TEXT,
    robot_instance                INTEGER,
    dataset_id                    TEXT,
    -- Caller capture timestamps (ns); routed window bounds live in memory.
    start_timestamp_ns            INTEGER,
    stop_timestamp_ns             INTEGER,
    expected_trace_count          INTEGER,
    expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
    progress_reported             TEXT    NOT NULL DEFAULT 'pending',
    -- Daemon wall-clock lifecycle timestamps.
    stopped_at                    DATETIME,
    cancelled_at                  DATETIME,
    -- Cloud-notify bookkeeping.
    backend_start_notified_at     DATETIME,
    backend_stop_notified_at      DATETIME,
    backend_cancel_notified_at    DATETIME,
    created_at                    DATETIME NOT NULL,
    last_updated                  DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id              TEXT PRIMARY KEY,
    recording_index       INTEGER NOT NULL,
    write_status          TEXT NOT NULL DEFAULT 'pending',
    registration_status   TEXT NOT NULL DEFAULT 'pending',
    upload_status         TEXT NOT NULL DEFAULT 'pending',
    data_type             TEXT,
    data_type_name        TEXT,
    path                  TEXT,
    bytes_written         INTEGER NOT NULL DEFAULT 0,
    total_bytes           INTEGER NOT NULL DEFAULT 0,
    bytes_uploaded        INTEGER NOT NULL DEFAULT 0,
    error_code            TEXT,
    error_message         TEXT,
    upload_session_uris   TEXT,
    created_at            DATETIME NOT NULL,
    last_updated          DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recordings_stopped_at
    ON recordings(stopped_at);
CREATE INDEX IF NOT EXISTS idx_recordings_cancelled_at
    ON recordings(cancelled_at);
CREATE INDEX IF NOT EXISTS idx_recordings_source
    ON recordings(robot_id, robot_instance, recording_index);
CREATE INDEX IF NOT EXISTS idx_recordings_start_notify
    ON recordings(recording_id, backend_start_notified_at);
-- `traces.recording_index` deliberately carries NO foreign key, and
-- `PRAGMA foreign_keys` is left at SQLite's OFF default. The recording → trace
-- cascade is hand-rolled in `SqliteStateStore::delete_recording_cascade` (two
-- DELETEs in one transaction) as the single, deliberate integrity mechanism:
-- it keeps the reaper's delete free of FK-ordering constraints and avoids
-- enabling the pragma on every pooled connection. If that cascade is ever
-- split or reordered, orphan trace rows become possible — keep both deletes in
-- one transaction.
--
-- No `idx_traces_recording_index` on `traces(recording_index)`: the composite
-- `idx_traces_recording_upload` below has `recording_index` as its leading
-- column, so SQLite already uses it for plain `WHERE recording_index = ?`
-- lookups. A separate single-column index would be pure write amplification.
CREATE INDEX IF NOT EXISTS idx_traces_recording_upload
    ON traces(recording_index, upload_status);
CREATE INDEX IF NOT EXISTS idx_traces_write_status
    ON traces(write_status);
CREATE INDEX IF NOT EXISTS idx_traces_registration_status
    ON traces(registration_status);
CREATE INDEX IF NOT EXISTS idx_traces_upload_status
    ON traces(upload_status);
