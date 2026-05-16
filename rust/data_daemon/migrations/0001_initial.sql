-- Initial schema for the Rust data daemon state store.
--
-- Mirrors the conceptual shape of the Python `tables.py` recordings + traces
-- schema. The column names and status enum strings here are part of the
-- behavioural contract integration tests rely on (see
-- `tests/integration/platform/data_daemon/shared/db_helpers.py`). Phase 3 only
-- creates the tables; later phases populate them.

CREATE TABLE IF NOT EXISTS recordings (
    recording_id                  TEXT PRIMARY KEY,
    org_id                        TEXT,
    expected_trace_count          INTEGER,
    expected_trace_count_reported INTEGER NOT NULL DEFAULT 0,
    trace_count                   INTEGER NOT NULL DEFAULT 0,
    uploaded_trace_count          INTEGER NOT NULL DEFAULT 0,
    progress_reported             TEXT    NOT NULL DEFAULT 'pending',
    stopped_at                    DATETIME,
    created_at                    DATETIME NOT NULL,
    last_updated                  DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id              TEXT PRIMARY KEY,
    recording_id          TEXT NOT NULL,
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
    created_at            DATETIME NOT NULL,
    last_updated          DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recordings_stopped_at
    ON recordings(stopped_at);

CREATE INDEX IF NOT EXISTS idx_traces_recording_id
    ON traces(recording_id);

CREATE INDEX IF NOT EXISTS idx_traces_recording_upload
    ON traces(recording_id, upload_status);

CREATE INDEX IF NOT EXISTS idx_traces_write_status
    ON traces(write_status);

CREATE INDEX IF NOT EXISTS idx_traces_registration_status
    ON traces(registration_status);

CREATE INDEX IF NOT EXISTS idx_traces_upload_status
    ON traces(upload_status);

CREATE INDEX IF NOT EXISTS idx_traces_next_retry_at
    ON traces(next_retry_at);
