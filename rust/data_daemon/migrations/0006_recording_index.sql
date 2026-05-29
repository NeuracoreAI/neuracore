-- Thin-shipper rewrite: the daemon owns recording identity.
--
-- The producer no longer mints or ships a recording id. The daemon assigns a
-- local `recording_index` (INTEGER PRIMARY KEY AUTOINCREMENT) the moment it
-- sees a `StartRecording` for a `(robot_id, robot_instance)` source, and a
-- separate, nullable cloud `recording_id` is filled in asynchronously by the
-- recording-start notifier (or minted on demand by the registration
-- coordinator when offline). The two columns are independent — no aliasing,
-- no fallback.
--
-- This rebuilds `recordings` (its PRIMARY KEY changes from the old
-- `recording_id TEXT` to `recording_index INTEGER`, which SQLite cannot do via
-- ALTER) and re-points `traces` from a `recording_id` foreign key to a
-- `recording_index` foreign key. This is a destructive rebuild: it is safe
-- only because the branch ships producer + daemon together and carries no
-- production data — dev databases and the integration suite both start from a
-- fresh state store.

DROP INDEX IF EXISTS idx_recordings_stopped_at;
DROP INDEX IF EXISTS idx_recordings_cancelled_at;
DROP INDEX IF EXISTS idx_traces_recording_id;
DROP INDEX IF EXISTS idx_traces_recording_upload;
DROP INDEX IF EXISTS idx_traces_write_status;
DROP INDEX IF EXISTS idx_traces_registration_status;
DROP INDEX IF EXISTS idx_traces_upload_status;
DROP INDEX IF EXISTS idx_traces_next_retry_at;

DROP TABLE IF EXISTS traces;
DROP TABLE IF EXISTS recordings;

CREATE TABLE recordings (
    recording_index               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Cloud handle. NULL until the recording-start notifier POSTs
    -- `/recording/start` (or registration mints one on demand when offline).
    recording_id                  TEXT,
    org_id                        TEXT,
    -- Source identity = (robot_id, robot_instance).
    robot_id                      TEXT,
    robot_instance                INTEGER,
    robot_name                    TEXT,
    dataset_id                    TEXT,
    dataset_name                  TEXT,
    -- Producer capture-clock window bounds (Unix nanoseconds).
    start_timestamp_ns            INTEGER,
    stop_timestamp_ns             INTEGER,
    expected_trace_count          INTEGER,
    expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
    uploaded_trace_count          INTEGER NOT NULL DEFAULT 0,
    progress_reported             TEXT    NOT NULL DEFAULT 'pending',
    -- Daemon wall-clock lifecycle timestamps.
    started_at                    DATETIME,
    stopped_at                    DATETIME,
    cancelled_at                  DATETIME,
    -- Cloud-notify bookkeeping.
    backend_start_notified_at     DATETIME,
    backend_start_failed_at       DATETIME,
    backend_stop_notified_at      DATETIME,
    created_at                    DATETIME NOT NULL,
    last_updated                  DATETIME NOT NULL
);

CREATE TABLE traces (
    trace_id              TEXT PRIMARY KEY,
    recording_index       INTEGER NOT NULL,
    write_status          TEXT NOT NULL DEFAULT 'pending',
    registration_status   TEXT NOT NULL DEFAULT 'pending',
    upload_status         TEXT NOT NULL DEFAULT 'pending',
    data_type             TEXT,
    data_type_name        TEXT,
    dataset_id            TEXT,
    dataset_name          TEXT,
    robot_name            TEXT,
    robot_id              TEXT,
    robot_instance        INTEGER,
    path                  TEXT,
    bytes_written         INTEGER NOT NULL DEFAULT 0,
    total_bytes           INTEGER NOT NULL DEFAULT 0,
    bytes_uploaded        INTEGER NOT NULL DEFAULT 0,
    error_code            TEXT,
    error_message         TEXT,
    num_upload_attempts   INTEGER NOT NULL DEFAULT 0,
    next_retry_at         DATETIME,
    upload_session_uris   TEXT,
    created_at            DATETIME NOT NULL,
    last_updated          DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recordings_stopped_at
    ON recordings(stopped_at);

CREATE INDEX IF NOT EXISTS idx_recordings_cancelled_at
    ON recordings(cancelled_at);

-- Source lookup: the dispatcher and tests find a source's recordings ordered
-- by recording_index.
CREATE INDEX IF NOT EXISTS idx_recordings_source
    ON recordings(robot_id, robot_instance, recording_index);

-- Pending cloud start-notify sweep predicate.
CREATE INDEX IF NOT EXISTS idx_recordings_start_notify
    ON recordings(recording_id, backend_start_notified_at, backend_start_failed_at);

CREATE INDEX IF NOT EXISTS idx_traces_recording_index
    ON traces(recording_index);

CREATE INDEX IF NOT EXISTS idx_traces_recording_upload
    ON traces(recording_index, upload_status);

CREATE INDEX IF NOT EXISTS idx_traces_write_status
    ON traces(write_status);

CREATE INDEX IF NOT EXISTS idx_traces_registration_status
    ON traces(registration_status);

CREATE INDEX IF NOT EXISTS idx_traces_upload_status
    ON traces(upload_status);

CREATE INDEX IF NOT EXISTS idx_traces_next_retry_at
    ON traces(next_retry_at);
